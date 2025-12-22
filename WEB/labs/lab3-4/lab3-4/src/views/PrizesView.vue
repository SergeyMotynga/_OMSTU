<template>
  <div>
    <h2>Нобелевские премии</h2>

    <div class="filters">
      <select v-model="selectedCategory" @change="onFilterChange">
        <option value="">Все категории</option>
        <option value="Physics">Физика</option>
        <option value="Chemistry">Химия</option>
        <option value="Physiology or Medicine">Медицина</option>
        <option value="Literature">Литература</option>
        <option value="Peace">Мир</option>
        <option value="Economic Sciences">Экономика</option>
      </select>
      <input v-model="selectedYear" type="number" placeholder="Год" @input="onFilterChange" />
      <input v-model="searchQuery" placeholder="Поиск" @input="onFilterChange" />
      <button @click="clearFilters">Очистить</button>
    </div>

    <div v-if="loading">Загрузка...</div>
    <DataTable v-else :columns="columns" :rows="rowsToShow" />

    <div class="pagination">
      <div class="pagination-info">
        Показано {{ startItem }}-{{ endItem }}
      </div>
      <div class="pagination-controls">
        <button :disabled="page === 1" @click="prevPage">Назад</button>
        <span>Стр. {{ page }}</span>
        <button :disabled="!hasMorePages" @click="nextPage">Вперёд</button>
      </div>
      <div class="items-per-page">
        <label>Записей на странице:</label>
        <select v-model="limit" @change="onItemsPerPageChange">
          <option :value="5">5</option>
          <option :value="10">10</option>
          <option :value="20">20</option>
          <option :value="50">50</option>
        </select>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import DataTable from '@/components/DataTable.vue'
import { nobelService } from '@/init/servicesNobel'
import type { Prize } from '@/services/features/nobel/codecs/PrizeCodec'
import type { PrizesResponse } from '@/services/features/nobel/codecs/ApiResponseCodecs'

const selectedCategory = ref('')
const selectedYear = ref('')
const searchQuery = ref('')
const page = ref(1)
const limit = ref(10)
const loading = ref(false)
const data = ref<Prize[]>([])
const totalItems = ref(0)
const hasMorePages = ref(false)

const columns = ['Год', 'Категория', 'Количество лауреатов']

const load = async () => {
  loading.value = true
  try {
    const params: Record<string, any> = {
      limit: limit.value,
      offset: (page.value - 1) * limit.value,
    }
    if (selectedYear.value) params.nobelPrizeYear = selectedYear.value

    const res: PrizesResponse = await nobelService.fetchPrizes(params)
    data.value = res.nobelPrizes
    totalItems.value = res.nobelPrizes.length
    hasMorePages.value = res.nobelPrizes.length === limit.value
  } catch (e) {
    console.error(e)
    data.value = []
    totalItems.value = 0
  } finally {
    loading.value = false
  }
}

const rowsToShow = computed(() => {
  let filtered = [...data.value]

  if (selectedCategory.value) {
    filtered = filtered.filter(p => p.category?.en === selectedCategory.value)
  }

  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(p =>
      p.category?.en?.toLowerCase().includes(query) ||
      p.awardYear?.includes(query)
    )
  }

  return filtered.map(p => ({
    'Год': p.awardYear ?? '–',
    'Категория': p.category?.en ?? '–',
    'Количество лауреатов': p.laureates?.length ?? 0,
  }))
})

const totalPages = computed(() => Math.ceil(totalItems.value / limit.value) || 1)
const startItem = computed(() => totalItems.value === 0 ? 0 : (page.value - 1) * limit.value + 1)
const endItem = computed(() => Math.min(page.value * limit.value, totalItems.value))

const onFilterChange = () => {
  page.value = 1
  load()
}

const clearFilters = () => {
  selectedCategory.value = ''
  selectedYear.value = ''
  searchQuery.value = ''
  page.value = 1
  load()
}

const onItemsPerPageChange = () => {
  page.value = 1
  load()
}

const nextPage = async () => {
  if (hasMorePages.value) {
    page.value++
    await load()
  }
}

const prevPage = async () => { 
  if (page.value > 1) {
    page.value--
    await load()
  }
}

load()
</script>

<style scoped>
.filters {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.pagination {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 20px;
  gap: 15px;
}

.pagination-controls {
  display: flex;
  gap: 10px;
  align-items: center;
}

.items-per-page {
  display: flex;
  gap: 5px;
  align-items: center;
}
</style>
