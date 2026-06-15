import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import DatasetsView from '@/views/DatasetsView.vue'
import RunsView from '@/views/RunsView.vue'
import RunDetailView from '@/views/RunDetailView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView,
  },
  {
    path: '/datasets',
    name: 'datasets',
    component: DatasetsView,
  },
  {
    path: '/runs',
    name: 'runs',
    component: RunsView,
  },
  {
    path: '/runs/:id',
    name: 'run-detail',
    component: RunDetailView,
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
