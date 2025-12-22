import { createRouter, createWebHistory } from 'vue-router'
import NobelView from '@/views/NobelView.vue'
import PrizesView from '@/views/PrizesView.vue'

const routes = [
  { path: '/', redirect: '/laureates' },
  { path: '/laureates', component: NobelView },
  { path: '/prizes', component: PrizesView }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router