<template>
  <TableComponent
    title="Список нобелевских премий"
    :headers="headers"
    :data="prizesData"
  />
</template>

<script>
import TableComponent from '../components/TableComponent.vue'

export default {
  name: 'PrizesView',
  components: {
    TableComponent
  },
  data() {
    return {
      headers: ['Категория', 'Дата вручения', 'Стоимость гранта'],
      prizesData: [
        { category: 'Физика', date: '10.12.2022', grant: '1 000 000 $' },
        { category: 'Химия', date: '10.12.2022', grant: '1 000 000 $' },
        { category: 'Медицина', date: '10.12.2022', grant: '1 000 000 $' }
      ]
    }
  }
}
</script>




